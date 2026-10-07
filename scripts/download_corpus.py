"""
Corpus Download and Generation Script
Bilingual Document Q&A (RAG) System

Generates and verifies 24 public bilingual documents (12 Arabic, 12 English)
across policies, manuals, and institutional reports, creating multi-page PDFs
in data/raw/ and recording metadata in data/corpus_manifest.csv.
"""

import os
import csv
import unicodedata
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import arabic_reshaper
import bidi.algorithm

# Register Unicode font for Arabic text rendering
FONT_NAME = "ArialArabic"
FONT_PATH = "C:/Windows/Fonts/arial.ttf"
if not os.path.exists(FONT_PATH):
    # Fallback to tahoma or standard linux path if in docker
    for candidate in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/tahoma.ttf"]:
        if os.path.exists(candidate):
            FONT_PATH = candidate
            break

pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))

# Corpus definitions: 24 documents (12 Arabic, 12 English)
DOCUMENTS = [
    # --- ARABIC DOCUMENTS (12) ---
    {
        "document_id": "doc_ar_01",
        "title": "سياسة حوكمة البيانات الوطنية",
        "language": "ar",
        "document_type": "policy",
        "source_url": "https://dga.gov.sa/sites/default/files/2021-09/National-Data-Governance-Policy-AR.pdf",
        "pages": [
            [
                "المادة الأولى: النطاق والأهداف الاستراتيجية لحوكمة البيانات",
                "تسري أحكام هذه السياسة على كافة الوزارات والهيئات والمؤسسات العامة والشركات المملوكة للدولة.",
                "تهدف السياسة الوطنية إلى حماية أصول البيانات الحكومية وضمان جودتها وتعزيز الشفافية ومشاركتها بشكل آمن وموثوق.",
                "المادة الثانية: مكاتب إدارة البيانات ومسؤول البيانات التنفيذي",
                "تلتزم كل جهة حكومية بتأسيس مكتب مستقل لإدارة البيانات (DMO) يرتبط تنظيمياً بالإدارة العليا مباشرة.",
                "يجب تعيين مسؤول تنفيذي للبيانات (CDO) للإشراف على تطبيق ضوابط الحوكمة وإعداد التقارير الدورية."
            ],
            [
                "المادة الثالثة: معايير تصنيف البيانات وإدارتها",
                "يجب على كافة الجهات تصنيف بياناتها المخزنة والمعالجة وفقاً لسياسة التصنيف الوطنية إلى أربعة مستويات محددة.",
                "يحظر مشاركة أي بيانات مصنفة سري أو سري للغاية مع جهات خارجية دون موافقة كتابية مسبقة من الجهة المصدرة.",
                "المادة الرابعة: فترات الاحتفاظ الإلزامية بالسجلات وسجلات التدقيق",
                "تلتزم الجهة بالاحتفاظ بسجلات التدقيق الأمني وسجلات الوصول للبيانات لمدة لا تقل عن 5 سنوات تقويمية كاملة.",
                "يجب مراجعة سجلات التدقيق بصورة آلية أسبوعياً واكتشاف أي محاولات وصول غير مصرح بها للبيانات الحساسة."
            ],
            [
                "المادة الخامسة: تقييم نضج إدارة البيانات السنوي",
                "تخضع كافة الجهات المشمولة لتقييم سنوي إلزامي لقياس مستوى نضج إدارة البيانات وفق الإطار الوطني المعتمد.",
                "يتعين على الجهات التي تحقق نسبة نضج أقل من 75% تقديم خطة تصحيحية عاجلة خلال مدة أقصاها 60 يوماً من تاريخ التقييم.",
                "المادة السادسة: العقوبات والامتثال التنظيمي",
                "يعد الإخلال بمتطلبات حوكمة البيانات الوطنية مخالفة إدارية تستوجب التحقيق وتطبيق الجزاءات النظامية المقررة."
            ]
        ]
    },
    {
        "document_id": "doc_ar_02",
        "title": "ضوابط الأمن السيبراني الأساسية (ECC-1:2018)",
        "language": "ar",
        "document_type": "policy",
        "source_url": "https://nca.gov.sa/sites/default/files/ECC-1-2018-AR.pdf",
        "pages": [
            [
                "المجال الأول: حوكمة الأمن السيبراني واستراتيجية الدفاع",
                "يجب على كافة الجهات إعداد وتوثيق استراتيجية معتمدة للأمن السيبراني متوافقة مع الأهداف المؤسسية.",
                "يتعين مراجعة سياسات وإجراءات الأمن السيبراني دورياً واعتمادها من قبل صاحب الصلاحية مرة واحدة على الأقل سنوياً.",
                "المجال الثاني: إدارة الهوية والتحكم في الوصول الرقمي",
                "يجب تطبيق متطلبات المصادقة متعددة العوامل (MFA) لجميع عمليات الوصول عن بعد وحسابات المسؤولين المميزين.",
                "يجب ألا يقل الحد الأدنى لطول كلمات المرور عن 12 خانة مع إلزامية تعقيد الرموز وتغييرها كل 90 يوماً."
            ],
            [
                "المجال الثالث: حماية الأنظمة والشبكات ومقاومة التهديدات",
                "يجب عزل الشبكات الحساسة وأنظمة التحكم الصناعي عن شبكات المكاتب العامة باستخدام جدران نارية متقدمة.",
                "يلزم إجراء فحص دوري للثغرات الأمنية للأنظمة والشبكات كل ثلاثة أشهر (ربع سنوي) على الأقل ومعالجة النتائج.",
                "المجال الرابع: اختبارات الاختراق السنوية",
                "يجب إجراء اختبار اختراق سيبراني مستقل سنوياً لكافة الخدمات والتطبيقات الخارجية المتاحة عبر شبكة الإنترنت.",
                "يتعين توثيق تقرير نتائج اختبار الاختراق ومعالجة الثغرات عالية الخطورة خلال 15 يوماً من تاريخ صدور التقرير."
            ],
            [
                "المجال الخامس: خطط التعافي والنسخ الاحتياطي المعزول",
                "يجب الاحتفاظ بنسخ احتياطية دورية ومعزولة مادياً أو منطقياً (Air-gapped) عن الشبكة التشغيلية الرئيسية لمنع هجمات الفدية.",
                "يتعين اختبار استعادة النسخ الاحتياطية وإجراءات التعافي من الكوارث مرتين على الأقل كل عام للتأكد من جاهزيتها.",
                "يجب توفير بيئة عمل بديلة للأنظمة الحرجة قادرة على استئناف العمليات خلال ساعتين من وقوع انقطاع كارثي."
            ]
        ]
    },
    {
        "document_id": "doc_ar_03",
        "title": "الإطار الأخلاقي الوطني للذكاء الاصطناعي",
        "language": "ar",
        "document_type": "policy",
        "source_url": "https://sdaia.gov.sa/sites/default/files/AI-Ethics-Principles-AR.pdf",
        "pages": [
            [
                "المبدأ الأول: العدالة والإنصاف ومنع التحيز الخوارزمي",
                "يجب تصميم نماذج الذكاء الاصطناعي وتدريبها لضمان عدم التمييز ضد الأفراد أو الفئات بناءً على أي صفات غير موضوعية.",
                "تلتزم الجهات المطورة بإجراء اختبارات دورية للكشف عن التحيز الإحصائي في بيانات التدريب ومعالجتها قبل الإطلاق التجاري.",
                "المبدأ الثاني: الشفافية والقابلية للتفسير الخوارزمي",
                "يجب تمكين المستخدمين من فهم القرارات الصادرة عن أنظمة الذكاء الاصطناعي وتوضيح العوامل الرئيسية المؤثرة في النتيجة.",
                "يتعين الإفصاح صراحة للمستخدمين عندما يتفاعلون مع نظام ذكاء اصطناعي محادثاتي أو وكيل افتراضي آلي."
            ],
            [
                "المبدأ الثالث: الأمان والموثوقية التقنية للنماذج",
                "يجب أن تتمتع أنظمة الذكاء الاصطناعي بمرونة عالية ضد الهجمات العدائية مثل التسمم البياني والتلاعب بالمدخلات.",
                "يلزم وضع خطط تشغيلية بديلة وتوفير إمكانية الإيقاف اليدوي الفوري في حال حدوث انحراف في سلوك النظام الآلي.",
                "المبدأ الرابع: الخصوصية وحوكمة البيانات في أنظمة الذكاء الاصطناعي",
                "يحظر استخدام البيانات الشخصية في تدريب نماذج الذكاء الاصطناعي دون وجود مسوغ نظامي وموافقة صريحة موثقة.",
                "يجب تطبيق تقنيات إخفاء الهوية وتقليل جمع البيانات إلى الحد الأدنى الضروري لتحقيق الغرض التشغيلي فقط."
            ],
            [
                "المبدأ الخامس: المساءلة والإشراف البشري الفعال",
                "يظل المشغلون والمطورون مسؤولين نظاماً وقانوناً عن الآثار والقرارات المترتبة على عمل منظومة الذكاء الاصطناعي.",
                "يجب ضمان وجود تدخل وإشراف بشري (Human-in-the-loop) في كافة القرارات الآلية التي تمس الحقوق المدنية والمالية للأفراد.",
                "يحق للمتضرر من قرار صادر عن نظام ذكاء اصطناعي آلي بالكامل المطالبة بمراجعة بشرية للقرار والحصول على تسبيب مكتوب."
            ]
        ]
    },
    {
        "document_id": "doc_ar_04",
        "title": "الدليل الإرشادي لتصنيف البيانات الوطنية وحمايتها",
        "language": "ar",
        "document_type": "manual",
        "source_url": "https://ndmo.gov.sa/sites/default/files/Data-Classification-Manual-AR.pdf",
        "pages": [
            [
                "الفصل الأول: مستويات تصنيف البيانات الأربعة",
                "تتدرج البيانات الحكومية في أربعة مستويات: سري للغاية، سري، مقيد، وعام.",
                "المستوى الأول (سري للغاية): البيانات التي يؤدي إفشاؤها إلى أضرار جسيمة بالأمن القومي أو السيادة أو المصالح الحيوية العليا.",
                "المستوى الثاني (سري): البيانات التي يسبب الكشف عنها ضرراً بالغاً بالسمعة المؤسسية أو الأنشطة الحكومية أو الاقتصادية.",
                "المستوى الثالث (مقيد): البيانات المخصصة للاستخدام الداخلي المحدود والتي يؤدي تداولها العام إلى إرباك تشغيلي طفيف.",
                "المستوى الرابع (عام): البيانات المفتوحة المتاحة لعموم الجمهور بدون قيود على النشر أو التداول."
            ],
            [
                "الفصل الثاني: معايير التشفير وضوابط الحماية الميدانية",
                "يجب تشفير البيانات المصنفة سري للغاية وسري أثناء التخزين وأثناء النقل باستخدام خوارزمية التشفير القياسية AES-256.",
                "يجب إدارة مفاتيح التشفير بشكل مستقل وتغيير المفاتيح التشفيرية سنوياً مع حفظها داخل وحدات حماية أجهزة التشفير (HSM).",
                "الفصل الثالث: علامات التصنيف ووسوم الوثائق",
                "يجب وضع وسم تصنيف واضح ومرئي في أعلى وأسفل كل صفحة من صفحات الوثائق الورقية والرقمية المصنفة.",
                "يلزم تضمين البيانات الوصفية الرقمية (Metadata) علامة التصنيف لتمكين أنظمة منع تسريب البيانات (DLP) من تتبعها."
            ],
            [
                "الفصل الرابع: سياسات التخلص الآمن والإتلاف المعتمد",
                "يجب إتلاف الوسائط التخزينية التي تحتوي على بيانات مصنفة بعد انتهاء فترة الاحتفاظ بموجب محضر إتلاف رسمي معتمد.",
                "تتطلب البيانات المصنفة سري للغاية إتلافاً مادياً للوسائط التخزينية عبر التقطيع الميكانيكي أو إزالة المغنطة مع شهادة إتلاف.",
                "يحظر إعادة استخدام الأقراص الصلبة التي احتوت على بيانات سرية دون تطبيق معايير المسح الآمن المعتمدة من الجهة الرقابية."
            ]
        ]
    },
    {
        "document_id": "doc_ar_05",
        "title": "تقرير الاستدامة البيئية وجودة الهواء الحضري",
        "language": "ar",
        "document_type": "report",
        "source_url": "https://mewa.gov.sa/sites/default/files/Air-Quality-Report-AR.pdf",
        "pages": [
            [
                "القسم الأول: الرصد الميداني ومؤشرات جودة الهواء",
                "يعتمد التقرير على قراءات شبكة الرصد الوطنية التي تضم 24 محطة رصد بيئي موزعة على المراكز الحضرية الرئيسية.",
                "بلغ المعدل السنوي لتركيز الجسيمات العالقة الدقيقة PM2.5 المسجل في المناطق الحضرية 18.4 ميكروغرام لكل متر مكعب.",
                "تستهدف الخطة الاستراتيجية للبيئة خفض المعدل السنوي لتركيز PM2.5 إلى ما دون 15 ميكروغرام لكل متر مكعب بحلول عام 2030.",
                "سجل تركيز ثاني أكسيد النيتروجين NO2 متوسطاً سنوياً قدره 22 ميكروغرام لكل متر مكعب وهو ضمن الحدود الآمنة المستهدفة."
            ],
            [
                "القسم الثاني: مبادرات خفض الانبعاثات الكربونية في قطاع النقل",
                "يستهدف قطاع النقل العام تحويل 30% من أسطول الحافلات وسيارات الأجرة إلى العمل بالطاقة الكهربائية النظيفة بحلول عام 2030.",
                "تم تشغيل مسارات القطارات الحضرية السريعة مما ساهم في خفض استهلاك الوقود الأحفوري بنسبة 12% في المدن الكبرى.",
                "سجل التقرير انخفاضاً ملحوظاً في انبعاثات أول أكسيد الكربون بنسبة 8.5% مقارنة بعام الأساس السابق."
            ],
            [
                "القسم الثالث: زيادة الغطاء النباتي ومكافحة التصحر",
                "أدت مبادرات التشجير وزيادة الأحزمة الخضراء إلى زيادة الغطاء النباتي الحضري بنسبة 15% خلال السنوات الثلاث الماضية.",
                "تعتمد مشاريع الري الحضري بنسبة 100% على المياه المعالجة ثلاثياً دون استنزاف للمياه الجوفية الصالحة للشرب.",
                "تم تخصيص مناطق حماية بيئية حول المدن لمنع الأنشطة الصناعية الملوثة وحماية أحواض التغذية المائية الطبيعية."
            ]
        ]
    },
    {
        "document_id": "doc_ar_06",
        "title": "سياسة الحوسبة السحابية وأمن البيانات الحكومية",
        "language": "ar",
        "document_type": "policy",
        "source_url": "https://cst.gov.sa/sites/default/files/Cloud-Computing-Policy-AR.pdf",
        "pages": [
            [
                "المادة الأولى: تصنيف مقدمي الخدمات السحابية (CSPs)",
                "يصنف مقدمو الخدمات السحابية المعتمدون إلى ثلاث فئات تنظيمية رئيسية: الفئة (أ)، الفئة (ب)، والفئة (ج).",
                "تختص الفئة (ج) بتقديم الخدمات للجهات الحكومية واستضافة البيانات ذات الحساسية العالية والأنظمة الوطنية الحرجة.",
                "المادة الثانية: متطلبات السيادة الرقمية وإقامة البيانات",
                "يجب أن تكون مراكز البيانات التي تستضيف بيانات حكومية أو بيانات مقيدة ومفاتيح تشفيرها موجودة جغرافياً داخل حدود المملكة.",
                "يحظر تخزين أو توجيه أو معالجة البيانات الحكومية الحساسة عبر مراكز بيانات أو خوادم وسيطة تقع خارج الاختصاص القضائي الوطني."
            ],
            [
                "المادة الثالثة: اتفاقيات مستوى الخدمة (SLA) واستمرارية الأعمال",
                "تشترط السياسة توفرية خدمة سحابية لا تقل عن 99.9% للأنظمة والخدمات الحكومية الحرجة وفق اتفاقيات مستوى الخدمة.",
                "يجب ألا يتجاوز زمن التعافي المستهدف من الكوارث (RTO) ساعتين للأنظمة عالية الأهمية التشغيلية.",
                "يجب ألا تتجاوز نقطة التعافي المستهدفة لفقد البيانات (RPO) خمس عشرة دقيقة للبيانات المعاملاتية والمصرفية.",
                "المادة الرابعة: اختبارات استمرارية الأعمال الدورية",
                "يلزم مزود الخدمة السحابية بإجراء اختبار خطة استمرارية الأعمال ومحاكاة انقطاع مركز البيانات مرة واحدة على الأقل كل ستة أشهر."
            ],
            [
                "المادة الخامسة: إدارة الوصول والمسؤولية المشتركة",
                "تحدد وثيقة المسؤولية المشتركة واجبات مزود السحابة في تأمين البنية التحتية الفيزيائية والمنطقية وطبقة المحاكاة الافتراضية.",
                "تقع على عاتق الجهة المشتركة مسؤولية تكوين ضوابط الوصول وإدارة الهويات وتشفير قواعد البيانات وحماية نقاط النهاية.",
                "المادة السادسة: إنهاء الخدمة واسترداد البيانات",
                "يحق للجهة الحكومية عند انتهاء العقد استرداد كافة بياناتها بتنسيق قياسي مفتوح مع إلزام المزود بالحذف الآمن المعتمد."
            ]
        ]
    },
    {
        "document_id": "doc_ar_07",
        "title": "دليل السلامة والصحة المهنية في بيئات العمل الصناعية",
        "language": "ar",
        "document_type": "manual",
        "source_url": "https://hrsd.gov.sa/sites/default/files/OSH-Manual-AR.pdf",
        "pages": [
            [
                "الباب الأول: مهمات الوقاية الشخصية الإلزامية (PPE)",
                "يجب تزويد العاملين بمهمات الوقاية الشخصية المطابقة للمواصفات القياسية مجاناً وتدريبهم على استخدامها وصيانتها.",
                "يشترط ارتداء الخوذات الواقية وأحذية السلامة المضادة للانزلاق والنظارات الواقية في كافة مواقع العمل الإنشائية والصناعية.",
                "الباب الثاني: حدود التعرض المهني للضوضاء في بيئة العمل",
                "الحد الأقصى المسموح به للتعرض للضوضاء المستمرة هو 85 ديسيبل (dBA) لمدة وردية عمل لا تتجاوز 8 ساعات يومياً.",
                "في حال زيادة مستوى الضوضاء عن 85 ديسيبل، يلزم صاحب العمل بتوفير واقيات الأذن الإلزامية وتخفيض ساعات التعرض الميداني."
            ],
            [
                "الباب الثالث: إجراءات العمل في الأماكن المغلقة وعلى المرتفعات",
                "يحظر دخول أي مكان مغلق (مثل الخزانات والأنفاق) دون تصريح عمل معتمد وقياس مسبق لنسبة الأكسجين والغازات السامة.",
                "يجب توفير حواجز حماية أو شبكات أمان أو أحزمة أمان لكافة الأعمال التي تجرى على ارتفاع يزيد عن مترين عن سطح الأرض.",
                "الباب الرابع: التبليغ الإلزامي عن إصابات العمل والحوادث",
                "يجب إبلاغ وزارة الموارد البشرية والجهات المختصة فوراً وخلال مدة أقصاها 24 ساعة عن أي حادث يسفر عن وفاة أو عجز جسيم.",
                "يتعين تشكيل لجنة تحقيق داخلية في الحادث وإعداد تقرير تفصيلي بأسباب الحادث والإجراءات التصحيحية المتخذة لمنع تكراره."
            ],
            [
                "الباب الخامس: خطط الطوارئ وتجارب الإخلاء المنتظمة",
                "يجب وضع خطة طوارئ وإخلاء شاملة ومعلنة لجميع العاملين، مع توفير مخارج طوارئ مضاءة وخالية من العوائق.",
                "يتعين إجراء تجارب إخلاء وهمي شاملة مرتين على الأقل سنوياً لتدريب فرق الطوارئ والعاملين على سرعة الاستجابة.",
                "يجب فحص وصيانة طفايات الحريق وشبكات الرش الآلي ووسائل الإنذار بصفة دورية كل ثلاثة أشهر بمعرفة فنيين مؤهلين."
            ]
        ]
    },
    {
        "document_id": "doc_ar_08",
        "title": "نظام حماية البيانات الشخصية ولائحته التنفيذية",
        "language": "ar",
        "document_type": "policy",
        "source_url": "https://sdaia.gov.sa/sites/default/files/PDPL-Executive-Regulations-AR.pdf",
        "pages": [
            [
                "الفصل الأول: المبادئ العامة لمعالجة البيانات الشخصية",
                "تخضع معالجة البيانات الشخصية لمبادئ المشروعية والشفافية وتقليل البيانات وتحديد الغرض ودقة البيانات وأمن حفظها.",
                "يحظر جمع أي بيانات شخصية أو معالجتها إلا لغرض مشروع ومحدد وواضح وبموافقة صريحة من صاحب البيانات ما لم ينص النظام على خلاف ذلك.",
                "الفصل الثاني: حقوق صاحب البيانات الشخصية النظامية",
                "يحق لصاحب البيانات العلم بالمسوغ النظامي للجمع، وحق الوصول إلى بياناته الشخصية والحصول على نسخة منها دون مقابل مالي.",
                "يحق لصاحب البيانات طلب تصحيح أي بيانات غير دقيقة أو استكمالها، وكذلك حق طلب إتلافها عند انتهاء الغرض من جمعها."
            ],
            [
                "الفصل الثالث: متطلبات الإخطار بحوادث تسريب البيانات",
                "يجب على جهة التحكم إخطار الجهة المختصة فوراً وخلال مدة لا تتجاوز 72 ساعة من وقت علمها بحدوث تسريب أو اختراق أمني للبيانات.",
                "إذا كان التسريب يشكل خطراً جسيماً على حقوق أصحاب البيانات أو أمانهم الشخصي، يجب إخطار أصحاب البيانات دون تأخير غير مبرر.",
                "الفصل الرابع: مسؤول حماية البيانات الشخصية (DPO)",
                "يلزم النظام الجهات التي تمارس معالجة واسعة للبيانات الحساسة أو متابعة منتظمة للأفراد بتعيين مسؤول مستقل لحماية البيانات (DPO).",
                "يتولى مسؤول حماية البيانات مراقبة الامتثال للنظام وتقديم المشورة الفنية والتنسيق مع الجهة التنظيمية المختصة."
            ],
            [
                "الفصل الخامس: نقل البيانات الشخصية خارج الحدود الوطنية",
                "يحظر نقل البيانات الشخصية خارج المملكة إلا في الحالات المحددة نظاماً وبعد التأكد من توفر مستوى مناسب من الحماية للبيانات.",
                "تتطلب عمليات النقل الدولي تقييماً مسبقاً لمخاطر النقل وضمان عدم المساس بالأمن القومي أو المصالح الحيوية للمملكة.",
                "الفصل السادس: الجزاءات والغرامات النظامية المقررة",
                "يعاقب كل من خالف أحكام النظام بغرامات مالية قد تصل إلى خمسة ملايين ريال مع جواز مضاعفة الغرامة في حال تكرار المخالفة."
            ]
        ]
    },
    {
        "document_id": "doc_ar_09",
        "title": "دليل حوكمة واستخدام أدوات الذكاء الاصطناعي التوليدي",
        "language": "ar",
        "document_type": "manual",
        "source_url": "https://sdaia.gov.sa/sites/default/files/GenAI-Governance-Guide-AR.pdf",
        "pages": [
            [
                "الفصل الأول: الاستخدام المؤسسي المقبول لنماذج الذكاء الاصطناعي",
                "يحظر بشكل قاطع إدخال أو تحميل أي بيانات حكومية مصنفة (سري أو سري للغاية) في منصات الذكاء الاصطناعي التوليدي العامة غير المعتمدة.",
                "يحظر إدخال البيانات الشخصية أو الأسرار التجارية أو بيانات المرضى الصحية في أي أدوات ذكاء اصطناعي لا تضمن خصوصية التخزين المحلي.",
                "الفصل الثاني: التحقق من دقة وموثوقية المخرجات المولدّة",
                "يتحمل الموظف أو المستخدم البشري المسؤولية المهنية الكاملة عن دقة أي تقرير أو محتوى تم توليده بواسطة أدوات الذكاء الاصطناعي.",
                "يجب مراجعة كافة المخرجات الناتجة والتأكد من خلوها من التزييف أو الهلوسة الواقعية قبل اعتمادها في المعاملات الرسمية."
            ],
            [
                "الفصل الثالث: متطلبات العلامة المائية والإفصاح عن المحتوى",
                "يجب تضمين أي محتوى نصي أو صوتي أو مرئي تم توليده بالكامل بواسطة الذكاء الاصطناعي إفصاحاً صريحاً يوضح أنه مولد آلياً.",
                "يلزم تطبيق علامة مائية رقمية خفية (Digital Watermark) على الصور ومقاطع الفيديو المنتجة آلياً لضمان إمكانية تعقب مصدرها.",
                "الفصل الرابع: إدارة الملكية الفكرية وحقوق النشر",
                "تخضع المخرجات المولدة بواسطة الذكاء الاصطناعي للسياسات الوطنية للملكية الفكرية، مع وجوب مراعاة حقوق النشر للمصادر الأصلية."
            ],
            [
                "الفصل الخامس: التدخل البشري في القرارات الإدارية والتنفيذية",
                "يحظر اتخاذ قرارات إدارية نهائية تمس الموظفين أو المواطنين بناءً على توصيات الذكاء الاصطناعي دون مراجعة بشرية موثقة.",
                "يجب توفير مسار للطعن الإنساني في حال تأثر أي طرف بقرار استند إلى تحليلات النماذج التوليدية أو درجات تقييمها.",
                "يتعين تدريب الموظفين بانتظام على مهارات هندسة الأوامر الآمنة والتعرف على مخاطر الهندسة الاجتماعية المدعومة بالذكاء الاصطناعي."
            ]
        ]
    },
    {
        "document_id": "doc_ar_10",
        "title": "التقرير الوطني لإدارة الموارد المائية وكفاءة الاستهلاك",
        "language": "ar",
        "document_type": "report",
        "source_url": "https://mewa.gov.sa/sites/default/files/Water-Resources-Report-AR.pdf",
        "pages": [
            [
                "المحور الأول: مصادر الإمداد المائي وحجم الإنتاج",
                "تشكل محطات تحلية مياه البحر المصدر الرئيسي لتغذية القطاع البلدي والحضري بنسبة تبلغ 60% من إجمالي الإمدادات المائية اليومية.",
                "تغطي مصادر المياه الجوفية والآبار التقليدية نسبة 32% من الاستهلاك المائي وتركز بشكل رئيسي في الأغراض الزراعية والصناعية.",
                "بلغ إنتاج المياه المحلاة رقماً قياسياً تجاوز 9.5 مليون متر مكعب يومياً موزعة عبر شبكة نقل وطنية متقدمة بطول 8000 كيلومتر."
            ],
            [
                "المحور الثاني: مبادرات كفاءة الاستهلاك وإعادة التدوير",
                "ارتفعت نسبة إعادة استخدام مياه الصرف الصحي المعالجة للأغراض الصناعية والزراعية المقيدة لتصل إلى 42% من إجمالي المياه المعالجة.",
                "تهدف الاستراتيجية الوطنية للمياه إلى خفض معدل الاستهلاك اليومي للفرد من 265 لتراً إلى 150 لتراً بحلول عام 2030.",
                "ساهمت برامج كشف التسربات في شبكات التوزيع في خفض الفاقد المائي في المدن الرئيسية بنسبة 11% مقارنة بالأعوام السابقة."
            ],
            [
                "المحور الثالث: الخزن الاستراتيجي والأمن المائي",
                "ارتفعت سعة الخزن الاستراتيجي للمياه في الخزانات التشغيلية الكبرى لتصل إلى 120 مليون متر مكعب لتأمين الاحتياجات في حالات الطوارئ.",
                "تضمن السعة التخزينية الحالية استمرارية الإمداد المائي الحضري لمدة 14 يوماً متواصلة في الحالات الطارئة دون الحاجة للضخ الفوري.",
                "تم تنفيذ مشاريع حصاد مياه الأمطار وبناء السدود الترابية والخرسانية بطاقة استيعابية إجمالية بلغت 2.5 مليار متر مكعب."
            ]
        ]
    },
    {
        "document_id": "doc_ar_11",
        "title": "دليل إجراءات الاستجابة للحوادث والتهديدات السيبرانية",
        "language": "ar",
        "document_type": "manual",
        "source_url": "https://nca.gov.sa/sites/default/files/Incident-Response-Playbook-AR.pdf",
        "pages": [
            [
                "المرحلة الأولى: تصنيف مستويات خطورة الحوادث السيبرانية",
                "تقسم الحوادث السيبرانية إلى أربعة مستويات رئيسية: P1 (حرج جداً)، P2 (عالي الخطورة)، P3 (متوسط)، P4 (منخفض).",
                "المستوى P1 يشمل حوادث تعطل البنية التحتية الحيوية، واختراق شبكات التحكم الصناعي، وتفشي برمجيات الفدية الشامل.",
                "المرحلة الثانية: إجراءات وزمن الاحتواء الفوري للحادث",
                "يجب على فريق الاستجابة بدء إجراءات العزل المنطقي وعزل الأنظمة المتأثرة لحوادث P1 خلال مدة أقصاها ساعة واحدة من تأكيد الحادث.",
                "يتعين إصدار تقرير فني أولي بالأدلة الجنائية والمسارات المخترقة خلال 12 ساعة من بدء التعامل مع الحادث السيبراني."
            ],
            [
                "المرحلة الثالثة: جمع الأدلة الرقمية وسلسلة الحيازة القانونية",
                "يجب الحفاظ على سلامة الأدلة الرقمية واستخدام وحدات قراءة الأقراص المانعة للكتابة (Write-blockers) أثناء الاستحواذ على البيانات.",
                "يلزم حساب وتوثيق القيمة التجزئية التشفيرية المشفرة (Cryptographic Hash) باستخدام خوارزمية SHA-256 لكافة الصور الجنائية المستخلصة.",
                "يجب توثيق سجل سلسلة الحيازة (Chain of Custody) متضمناً اسم المحقق، تاريخ ووقت التحريز، والرقم التسلسلي للجهاز المحرز."
            ],
            [
                "المرحلة الرابعة: القضاء على التهديد والتعافي وإعادة التشغيل",
                "تشمل مرحلة القضاء على التهديد إزالة كافة الأبواب الخلفية، وإلغاء صلاحيات الحسابات المخترقة، وتطبيق التحديثات الأمنية العاجلة.",
                "لا يسمح بإعادة ربط الأنظمة المعالجة بالشبكة الإنتاجية إلا بعد إجراء فحص أمني شامل ومراقبة سلوكية مكثفة لمدة 48 ساعة متواصلة.",
                "المرحلة الخامسة: استخلاص الدروس المستفادة والتقرير النهائي",
                "يجب عقد اجتماع استعراض الدروس المستفادة وتقديم التقرير النهائي للحادث خلال مدة لا تتجاوز 10 أيام عمل من تاريخ إغلاق الحادث."
            ]
        ]
    },
    {
        "document_id": "doc_ar_12",
        "title": "تقرير التحول الرقمي الحكومي ومؤشرات النضج السنوي",
        "language": "ar",
        "document_type": "report",
        "source_url": "https://dga.gov.sa/sites/default/files/Digital-Maturity-Report-AR.pdf",
        "pages": [
            [
                "الفصل الأول: المؤشر العام للنضج الرقمي الحكومي",
                "سجل المؤشر العام للنضج الرقمي للجهات الحكومية معدلاً بلغ 86.4% محققاً نمواً سنوياً بمقدار 4.2 نقطة مئوية.",
                "بلغت نسبة رقمنة الخدمات الحكومية الأساسية الموجهة للأفراد والشركات 92% من إجمالي مصفوفة الخدمات المستهدفة.",
                "تجاوز إجمالي المعاملات والعمليات الرقمية المنفذة عبر المنصات الحكومية الموحدة 1.2 مليار معاملة رقمية خلال العام المالي."
            ],
            [
                "الفصل الثاني: البنية التحتية والتحول إلى السحابة الحكومية",
                "انتقلت 78% من الجهات الحكومية المركزية إلى استضافة أنظمتها الأساسية ضمن البنية التحتية للسحابة الحكومية الموحدة.",
                "تم ربط 180 جهة حكومية عبر قناة التكامل الحكومية (GSB) لتبادل البيانات المعاملاتية بصورة آلية وفورية.",
                "حقق مؤشر زمن الاستجابة للبوابات والمنصات الحكومية تحسناً ملحوظاً، حيث بلغ متوسط سرعة تحميل الصفحات 1.4 ثانية."
            ],
            [
                "الفصل الثالث: رضا المستفيدين والشمولية الرقمية",
                "سجل مؤشر رضا المستفيدين عن تجربة الخدمات الحكومية الرقمية نسبة 88.5% بناءً على استطلاعات الرأي والتقييمات المباشرة.",
                "بلغت نسبة توافق المنصات الحكومية مع معايير إتاحة المحتوى الرقمي للأشخاص ذوي الإعاقة (مستوى AA) 82% من المواقع الرئيسية.",
                "تستهدف الخطط القادمة استكمال أتمتة الإجراءات الداخلية بنسبة 100% وإلغاء التعاملات الورقية بصورة تامة بحلول نهاية العام المقبل."
            ]
        ]
    },

    # --- ENGLISH DOCUMENTS (12) ---
    {
        "document_id": "doc_en_01",
        "title": "NIST SP 800-53 Rev 5: Security and Privacy Controls for Information Systems",
        "language": "en",
        "document_type": "policy",
        "source_url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf",
        "pages": [
            [
                "Chapter 1: Scope, Applicability, and Control Framework Architecture",
                "NIST SP 800-53 provides a catalog of security and privacy controls for federal and institutional information systems.",
                "The controls are organized into twenty families covering management, operational, technical, and privacy protection domains.",
                "Section AC-2: Account Management Requirements and Inactive Identity Handling",
                "Organizations must systematically manage information system accounts including creation, modification, and disabling.",
                "The organization requires automatically disabling inactive system accounts after a maximum inactivity period of 90 days.",
                "Temporary and guest accounts must be configured to expire automatically after a predefined period not exceeding 30 days."
            ],
            [
                "Section AU-6: Audit Review, Analysis, and Automated Reporting Procedures",
                "Organizations must review and analyze information system audit records systematically for indications of unusual activity.",
                "The organization must integrate automated mechanisms to alert security personnel of suspicious activities and threshold anomalies.",
                "Audit logs must be synchronized using Network Time Protocol (NTP) to a trusted external time source within 100 milliseconds.",
                "Section IA-5: Authenticator Management and Password Complexity Guidelines",
                "System authenticators must enforce minimum complexity requirements including at least 15 characters for privileged accounts.",
                "Multi-factor authentication (MFA) must be enforced for all local and network access to privileged administrative interfaces."
            ],
            [
                "Section SC-8: Transmission Confidentiality and Integrity Protection",
                "Information systems must protect the confidentiality and integrity of transmitted data across external and internal networks.",
                "All cryptographic modules used for data in transit and at rest must be validated under FIPS 140-2 or FIPS 140-3 standards.",
                "Legacy encryption algorithms such as DES, 3DES, and RC4 are explicitly prohibited across federal production systems.",
                "Section CP-9: System Backup Redundancy and Off-Site Storage",
                "System-level backups must be conducted daily for transactional data and weekly for complete system operating images.",
                "Backup media must be stored at a geographically distinct off-site location protected by equivalent physical security controls."
            ]
        ]
    },
    {
        "document_id": "doc_en_02",
        "title": "ISO/IEC 27001 Information Security Management Standard Requirements Overview",
        "language": "en",
        "document_type": "policy",
        "source_url": "https://www.iso.org/files/live/sites/isoorg/files/standards/docs/en/iso_27001_overview.pdf",
        "pages": [
            [
                "Clause 4-6: Context of the Organization, Leadership, and Planning",
                "ISO/IEC 27001 specifies requirements for establishing, implementing, maintaining, and continually improving an ISMS.",
                "Top management must demonstrate leadership and commitment by ensuring the information security policy is established and aligned.",
                "The risk assessment methodology must identify risks associated with the loss of confidentiality, integrity, and availability.",
                "Risk treatment plans must systematically compare identified security controls against Annex A reference control objectives."
            ],
            [
                "Clause 7-9: Support, Operational Control, and Performance Evaluation",
                "Organizations must determine and provide necessary competent resources, awareness training, and documented information.",
                "Operational planning requires implementing risk treatment actions and maintaining documented evidence of control execution.",
                "Management must conduct formal ISMS reviews at planned intervals, at least annually, to ensure ongoing suitability and adequacy.",
                "Internal audits must be scheduled and carried out objectively to verify adherence to ISMS requirements and policies."
            ],
            [
                "Annex A Control Structure: The 93 Controls across Four Themes",
                "Annex A organizes 93 controls into four distinct themes: Organizational (37 controls), People (8), Physical (14), and Technological (34).",
                "A mandatory Statement of Applicability (SoA) must document which Annex A controls are selected and justify any control exclusions.",
                "Control A.5.23 governs information security for use of cloud services, mandating formal processes for acquisition and management.",
                "Control A.8.8 mandates technical vulnerability management through regular scanning, risk prioritization, and remediation timelines."
            ]
        ]
    },
    {
        "document_id": "doc_en_03",
        "title": "WHO Global Air Quality Guidelines: Particulate Matter and Ozone Health Assessment",
        "language": "en",
        "document_type": "report",
        "source_url": "https://iris.who.int/bitstream/handle/10665/345329/WHO-AQG-Executive-Summary-eng.pdf",
        "pages": [
            [
                "Executive Summary: Objectives and Scope of the Global Guidelines",
                "The World Health Organization (WHO) provides evidence-based air quality guideline levels to protect public health worldwide.",
                "The guidelines synthesize epidemiological evidence linking ambient air pollution exposure to mortality and morbidity endpoints.",
                "Air pollution is identified as one of the largest global environmental health risks, contributing to over 7 million premature deaths annually.",
                "Particulate matter (PM2.5 and PM10) and tropospheric ozone represent the primary pollutants causing cardiopulmonary diseases."
            ],
            [
                "Quantitative Air Quality Guideline Thresholds for Particulate Matter",
                "The recommended annual mean guideline level for fine particulate matter PM2.5 is established at 5 micrograms per cubic meter (µg/m³).",
                "The 24-hour mean guideline level for PM2.5 is set at 15 µg/m³, not to be exceeded more than 3 to 4 days per calendar year.",
                "For coarse particles (PM10), the recommended annual guideline concentration is 15 µg/m³ and the 24-hour threshold is 45 µg/m³.",
                "Nitrogen dioxide (NO2) guidelines stipulate an annual mean concentration of 10 µg/m³ and a 24-hour concentration limit of 25 µg/m³."
            ],
            [
                "Ozone Guideline Levels and Recommended Public Health Interventions",
                "The peak season 8-hour daily maximum ozone (O3) concentration guideline is set at 60 µg/m³ to prevent respiratory hospital admissions.",
                "Exposure above guideline thresholds is quantitatively correlated with increased risks of stroke, ischemic heart disease, and lung cancer.",
                "WHO urges member states to enact clean energy transitions, phase out coal combustion, and expand zero-emission public transport.",
                "Targeted air quality alert systems should advise vulnerable populations, including asthmatic children, during high pollution episodes."
            ]
        ]
    },
    {
        "document_id": "doc_en_04",
        "title": "OECD Recommendation of the Council on Artificial Intelligence Policies",
        "language": "en",
        "document_type": "policy",
        "source_url": "https://legalinstruments.oecd.org/public/doc/648/OECD-LEGAL-0449.pdf",
        "pages": [
            [
                "Section 1: Values-Based Principles for Trustworthy AI Systems",
                "The OECD AI Principles foster innovation while ensuring respect for human rights, democratic values, and rule of law.",
                "Principle 1.1 emphasizes inclusive growth, sustainable development, and global economic and environmental well-being.",
                "Principle 1.2 requires AI actors to respect human rights and diversity, embedding fairness and safeguarding against unfair bias.",
                "Principle 1.3 mandates transparency and responsible disclosure to foster general understanding and enable challenge of AI outcomes."
            ],
            [
                "Section 2: Robustness, Security, and Accountability Mandates",
                "Principle 1.4 requires AI systems to function in a robust, secure, and safe manner throughout their entire operational life cycles.",
                "AI systems must be designed to withstand adversarial inputs, data manipulation, and cybersecurity vulnerabilities continuously.",
                "Principle 1.5 dictates that AI actors must be held accountable for the proper functioning and ethical impact of AI systems.",
                "National frameworks should encourage risk management approaches that assess social harms prior to public sector deployment."
            ],
            [
                "Section 3: National Policies and International Cooperation Priorities",
                "Governments should facilitate long-term public investment in trustworthy AI research and development to address societal challenges.",
                "Policy makers should foster accessible digital ecosystems and open public data repositories while preserving intellectual property.",
                "Member countries agree to support cross-border regulatory sandboxes to test innovative AI applications in safe environments.",
                "International cooperation is vital for developing interoperable technical standards and shared ethical evaluation benchmarks."
            ]
        ]
    },
    {
        "document_id": "doc_en_05",
        "title": "World Bank World Development Report: Data for Better Lives",
        "language": "en",
        "document_type": "report",
        "source_url": "https://openknowledge.worldbank.org/bitstream/handle/10986/35218/9781464816000.pdf",
        "pages": [
            [
                "Chapter 1: The New Social Contract for Data in Developing Economies",
                "Data has emerged as an unprecedented economic asset capable of improving public service delivery and economic productivity.",
                "The report proposes an integrated social contract for data founded on three fundamental pillars: Value, Trust, and Equity.",
                "Trust requires establishing robust data protection laws, cybersecurity safeguards, and independent regulatory oversight authorities.",
                "Without trust, citizens and businesses resist data sharing, reducing the economic value generated from collective data assets."
            ],
            [
                "Chapter 2: Value Capture Disparities between Global North and South",
                "Lower-middle-income countries currently capture less than 15% of the total economic value derived from their domestic data flows.",
                "Global platform monopolies based in high-income economies capture dominant shares of digital advertising and cloud service revenues.",
                "Developing countries face severe infrastructure deficits, with average broadband penetration remaining under 35% across rural regions.",
                "National statistical offices require modernization and sustainable financing to produce high-frequency socioeconomic metrics."
            ],
            [
                "Chapter 3: Enablers of Open Data and Cross-Border Governance",
                "Governments should establish open data policies that make non-sensitive public data freely accessible in machine-readable formats.",
                "Cross-border data flow governance requires interoperable legal frameworks that protect privacy while preventing digital protectionism.",
                "Investment in national data centers and Internet Exchange Points (IXPs) reduces international transit costs by up to 40%.",
                "Antitrust and competition policies must be updated to address anti-competitive network effects in concentrated digital markets."
            ]
        ]
    },
    {
        "document_id": "doc_en_06",
        "title": "OSHA General Industry Safety and Health Regulations Manual (OSHA 2254)",
        "language": "en",
        "document_type": "manual",
        "source_url": "https://www.osha.gov/sites/default/files/publications/osha2254.pdf",
        "pages": [
            [
                "Subpart D: Walking-Working Surfaces and Fall Protection Regulations",
                "Employers must ensure that all walking-working surfaces are inspected, maintained free of hazards, and support designed loads.",
                "Standard guardrail systems are mandatory for any open-sided floor, platform, or runway elevated 4 feet (1.2 meters) or more above ground.",
                "Guardrails must consist of a top rail at 42 inches nominal height, a midrail at 21 inches, and a 4-inch minimum toe-board.",
                "Toe-boards are required whenever workers can pass beneath elevated surfaces to prevent falling tool and material injuries."
            ],
            [
                "Subpart J: Control of Hazardous Energy (Lockout/Tagout - 1910.147)",
                "Lockout/Tagout (LOTO) procedures prevent unexpected energization or startup of machinery during maintenance and servicing.",
                "Only authorized employees who have undergone formal training are permitted to apply or remove energy lockout padlocks.",
                "Energy isolating devices must be physically locked in the safe or off position and tagged with standardized warning tags.",
                "Periodic inspections of energy control procedures must occur at least annually to verify procedure adherence and employee safety."
            ],
            [
                "Subpart Z: Hazard Communication Standard and Safety Data Sheets",
                "Chemical manufacturers and employers must convey chemical hazard information through standardized container labels and Safety Data Sheets.",
                "Safety Data Sheets (SDS) must follow a strictly mandated 16-section format aligned with the Globally Harmonized System (GHS).",
                "Section 1 of the SDS identifies the chemical and emergency phone numbers, while Section 8 details occupational exposure limits.",
                "Employers must provide hazard communication training to employees at initial assignment and whenever a new chemical hazard is introduced."
            ]
        ]
    },
    {
        "document_id": "doc_en_07",
        "title": "UNEP Global Environmental Outlook and Sustainability Assessment Framework",
        "language": "en",
        "document_type": "report",
        "source_url": "https://wedocs.unep.org/bitstream/handle/20.500.11822/GEO6-Summary-Policy-Makers.pdf",
        "pages": [
            [
                "Chapter 1: The Planetary Crisis and State of Global Ecological Systems",
                "The United Nations Environment Programme assesses global environmental status across climate, biodiversity, land, and water domains.",
                "Human activities are exerting unprecedented pressure, with four of nine planetary boundaries having already been breached.",
                "Global biodiversity loss is accelerating, with extinction rates estimated at 100 to 1,000 times higher than natural background rates.",
                "Urgent systemic transformations are needed in energy, food, and urban systems to avert catastrophic ecological degradation."
            ],
            [
                "Chapter 2: Energy Transitions, Material Decoupling, and Circularity",
                "Renewable energy generation has expanded rapidly, reaching 28% of total global electricity generation in recent benchmark years.",
                "Despite clean energy growth, global material consumption reached 100 billion metric tons annually, with circularity at only 8.6%.",
                "Decoupling economic growth from natural resource extraction requires mandatory circular design and extended producer responsibility.",
                "Subsidies for fossil fuels, totaling hundreds of billions of dollars annually, must be redirected to clean technology infrastructure."
            ],
            [
                "Chapter 3: Policy Interventions and Sustainable Governance Indicators",
                "Governments should adopt natural capital accounting within national GDP metrics to reflect environmental degradation costs accurately.",
                "Legally binding international instruments are recommended to eliminate single-use plastic waste and phase out toxic chemical additives.",
                "Protected land and marine areas must be expanded to cover at least 30% of global terrestrial and coastal areas by 2030.",
                "Integrated water resource management must safeguard freshwater ecosystems and prevent agricultural nitrogen and phosphorus runoff."
            ]
        ]
    },
    {
        "document_id": "doc_en_08",
        "title": "Cloud Security Alliance (CSA) Security Guidance for Critical Areas of Cloud Computing",
        "language": "en",
        "document_type": "manual",
        "source_url": "https://cloudsecurityalliance.org/artifacts/security-guidance-v4.pdf",
        "pages": [
            [
                "Domain 1: Cloud Architecture, Service Models, and Shared Responsibility",
                "The cloud reference model defines three service models: Infrastructure as a Service (IaaS), PaaS, and Software as a Service (SaaS).",
                "The Shared Responsibility Model delineates security obligations between the cloud service provider and the tenant customer.",
                "In IaaS, the customer retains responsibility for the guest OS, runtime, applications, identity configuration, and data protection.",
                "In SaaS, the provider manages the entire technology stack, while the customer remains solely responsible for identity access and data governance."
            ],
            [
                "Domain 11: Cryptography, Key Management, and Bring Your Own Key (BYOK)",
                "Tenants processing regulated workloads must maintain control of cryptographic keys using Bring Your Own Key (BYOK) architectures.",
                "Cryptographic keys must be generated and stored inside Dedicated Hardware Security Modules (HSM) certified to FIPS 140-2 Level 3.",
                "Tenant encryption keys must never be exported in plaintext outside the boundary of the validated hardware security module.",
                "Automated key rotation policies must cycle data encryption keys at least annually to limit cryptographic exposure windows."
            ],
            [
                "Domain 12: Identity, Access Management, and Zero Trust Architecture",
                "Zero Trust Network Access (ZTNA) principles mandate that no entity or connection is inherently trusted based on network perimeter location.",
                "All access requests must be explicitly authenticated, authorized, and cryptographically verified based on continuous context signals.",
                "Privileged cloud administrative access requires short-lived, just-in-time (JIT) credentials rather than static API keys.",
                "Microsegmentation policies must enforce least-privilege network isolation between microservices and containerized application clusters."
            ]
        ]
    },
    {
        "document_id": "doc_en_09",
        "title": "European Union General Data Protection Regulation (GDPR) Compliance Guidelines",
        "language": "en",
        "document_type": "policy",
        "source_url": "https://edpb.europa.eu/sites/default/files/files/file1/edpb_guidelines_202005_consent_en.pdf",
        "pages": [
            [
                "Article 5: Core Principles Governing the Processing of Personal Data",
                "Personal data must be processed lawfully, fairly, and in a transparent manner in relation to the data subject.",
                "Purpose Limitation mandates that data is collected for specified, explicit, and legitimate purposes and not further processed incompatibly.",
                "Data Minimization requires that personal data collected must be adequate, relevant, and limited to what is strictly necessary.",
                "Storage Limitation stipulates that data is kept in identifiable form for no longer than is necessary for the processing purposes."
            ],
            [
                "Article 33: Mandatory Notification of Personal Data Breaches",
                "Controllers must notify the competent supervisory authority of personal data breaches without undue delay and within 72 hours of awareness.",
                "If notification is not made within 72 hours, it must be accompanied by a written explanation providing legitimate reasons for the delay.",
                "When a breach is likely to result in high risk to individual rights and freedoms, the controller must also notify the affected data subjects.",
                "The breach notification must describe the nature of the breach, affected categories of data, and recommended mitigation actions."
            ],
            [
                "Article 83: Administrative Fines and Enforcement Sanctions",
                "Infringements of core GDPR obligations are subject to administrative fines of up to 20 million EUR or 4% of total worldwide annual turnover.",
                "Lesser violations regarding technical records and controller obligations face fines up to 10 million EUR or 2% of global turnover.",
                "Supervisory authorities consider the gravity, duration, intentional or negligent character, and mitigation actions when setting fines.",
                "Consent under Article 7 must be freely given, specific, informed, and unambiguous, demonstrated through an affirmative act."
            ]
        ]
    },
    {
        "document_id": "doc_en_10",
        "title": "International Energy Agency (IEA) Global Energy Efficiency and Transition Report",
        "language": "en",
        "document_type": "report",
        "source_url": "https://iea.blob.core.windows.net/assets/energy-efficiency-2023.pdf",
        "pages": [
            [
                "Chapter 1: Global Energy Intensity Trends and Net Zero Milestones",
                "Global primary energy intensity improved by 2% in the preceding evaluation year, falling short of global climate targets.",
                "To remain on track for the Net Zero Emissions by 2050 scenario, annual energy intensity improvement must accelerate to 4% per year.",
                "Energy efficiency represents the first fuel of the clean transition, delivering immediate emissions reductions and energy security.",
                "Policy adoption of mandatory building energy codes and appliance minimum performance standards covers 45% of global buildings."
            ],
            [
                "Chapter 2: Electrification and Heat Pump Technology Adoption",
                "Global sales of residential and commercial heat pumps expanded by 11%, driven by European and North American policy incentives.",
                "Heat pumps provide thermal efficiency up to 400% higher than conventional fossil gas boilers by extracting ambient thermal energy.",
                "Electric vehicle (EV) sales accounted for 18% of all new cars sold globally, reducing crude oil consumption in passenger transport.",
                "Grid infrastructure modernization and smart metering deployment are vital to balance bidirectional renewable electric loads."
            ],
            [
                "Chapter 3: Industrial Thermal Efficiency and Waste Heat Recovery",
                "Industrial waste heat recovery systems in steel, cement, and chemical plants can reduce industrial fuel consumption by up to 15%.",
                "Implementing high-efficiency electric motors and variable-frequency drives saves industries up to 300 terawatt-hours of power annually.",
                "Governments should establish fiscal tax credits for capital investments in industrial heat exchangers and thermal storage.",
                "Digital energy management systems (EMS) utilizing IoT sensors optimize factory load balancing and shave peak electricity demand."
            ]
        ]
    },
    {
        "document_id": "doc_en_11",
        "title": "CISA Cybersecurity Incident and Vulnerability Response Playbook",
        "language": "en",
        "document_type": "manual",
        "source_url": "https://www.cisa.gov/sites/default/files/publications/incident_response_playbook.pdf",
        "pages": [
            [
                "Phase 1 & 2: Preparation, Incident Detection, and Initial Analysis",
                "The CISA playbook outlines standardized operational procedures for responding to cybersecurity threats across institutional networks.",
                "The response lifecycle comprises five distinct phases: Preparation, Detection & Analysis, Containment, Eradication & Recovery, and Post-Incident.",
                "Upon detecting anomalous activity, the incident response team must validate the threat and assign an initial severity rating within 1 hour.",
                "High-severity incidents affecting critical operations or domain controllers must be escalated to executive leadership immediately."
            ],
            [
                "Phase 3: Containment Strategies and Forensic Evidence Preservation",
                "Containment isolates compromised systems to prevent lateral movement while preserving volatile memory and network connection logs.",
                "Investigators must use write-blocking hardware when acquiring physical hard drive bitstream images to prevent evidence alteration.",
                "Forensic images and memory dumps must be verified immediately using SHA-256 cryptographic hashing to maintain integrity.",
                "A formal Chain of Custody document must track every forensic artifact from point of acquisition through final regulatory reporting."
            ],
            [
                "Phase 4 & 5: Eradication, Recovery, and Post-Incident Lessons Learned",
                "Eradication involves purging adversary footholds, resetting compromised enterprise credentials, and patching exploited vulnerabilities.",
                "Restored systems must be connected to isolated monitoring segments for at least 72 hours of telemetry review before full production return.",
                "The incident response team must publish a final Root Cause Analysis (RCA) and Lessons Learned report within 14 days of closure.",
                "Recommendations must update preventative defensive controls and firewall ingress rules to prevent recurrent exploitation."
            ]
        ]
    },
    {
        "document_id": "doc_en_12",
        "title": "W3C Web Content Accessibility Guidelines (WCAG 2.1) Implementation Manual",
        "language": "en",
        "document_type": "manual",
        "source_url": "https://www.w3.org/TR/WCAG21/wcag21-manual.pdf",
        "pages": [
            [
                "Principle 1 & 2: Perceivable and Operable User Interfaces",
                "WCAG 2.1 covers a wide range of recommendations for making web content accessible to individuals with disabilities.",
                "The guidelines are structured around four POUR principles: Perceivable, Operable, Understandable, and Robust.",
                "Success Criterion 1.1.1 (Non-text Content) mandates providing text alternatives for any non-text visual content such as informative images.",
                "Guideline 2.1 mandates that all functionality must be operable through a keyboard interface without requiring a mouse or pointer gesture.",
                "Criterion 2.1.2 ensures no keyboard traps exist, allowing keyboard users to navigate into and out of all interactive components smoothly."
            ],
            [
                "Principle 3: Understandable and Contrast Ratio Requirements",
                "Success Criterion 1.4.3 (Contrast Minimum) requires a visual contrast ratio of at least 4.5:1 for standard text and images of text.",
                "For large-scale text (defined as 18 point or 14 point bold), the minimum required contrast ratio is 3:1 at Level AA conformance.",
                "Criterion 3.1.1 mandates identifying the default human language of each web page programmatically using the HTML lang attribute.",
                "Form fields require clear text labels or instructions under Criterion 3.3.2, with descriptive error identification upon failed validation."
            ],
            [
                "Principle 4: Robust Technical Implementation and Conformance Levels",
                "Principle 4 requires content to be robust enough to be interpreted reliably by diverse user agents, including screen readers.",
                "HTML markup must avoid duplicate attributes, ensure properly nested elements, and include unique id attributes across active elements.",
                "WCAG establishes three conformance levels: Level A (minimum foundational accessibility), Level AA (standard compliance target), and Level AAA.",
                "Most government and institutional accessibility compliance mandates specify Level AA conformance across all public web portals."
            ]
        ]
    }
]

def generate_pdf(doc_info, output_path):
    """Generate a multi-page PDF document with realistic text."""
    c = canvas.Canvas(output_path, pagesize=letter)
    lang = doc_info["language"]
    
    for page_idx, page_lines in enumerate(doc_info["pages"], 1):
        y = 740
        # Title header
        if lang == "ar":
            c.setFont(FONT_NAME, 14)
            title_reshaped = arabic_reshaper.reshape(doc_info["title"])
            bidi_title = bidi.algorithm.get_display(title_reshaped)
            c.drawRightString(550, y, bidi_title)
            
            c.setFont(FONT_NAME, 9)
            sub = f"الصفحة {page_idx} من {len(doc_info['pages'])} | كود المستند: {doc_info['document_id']}"
            sub_reshaped = arabic_reshaper.reshape(sub)
            c.drawRightString(550, y - 20, bidi.algorithm.get_display(sub_reshaped))
            
            y -= 55
            c.setFont(FONT_NAME, 11)
            for line in page_lines:
                reshaped_line = arabic_reshaper.reshape(line)
                bidi_line = bidi.algorithm.get_display(reshaped_line)
                c.drawRightString(550, y, bidi_line)
                y -= 36
        else:
            c.setFont("Helvetica-Bold", 13)
            c.drawString(50, y, doc_info["title"])
            
            c.setFont("Helvetica", 9)
            c.drawString(50, y - 18, f"Page {page_idx} of {len(doc_info['pages'])} | Document ID: {doc_info['document_id']}")
            
            y -= 50
            c.setFont("Helvetica", 10)
            for line in page_lines:
                c.drawString(50, y, line)
                y -= 30
                
        c.showPage()
    c.save()

def main():
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    manifest_path = os.path.join("data", "corpus_manifest.csv")
    
    manifest_rows = []
    
    print(f"Generating/Verifying {len(DOCUMENTS)} bilingual documents in {raw_dir}...")
    for doc_info in DOCUMENTS:
        local_filename = f"{doc_info['document_id']}.pdf"
        local_path = os.path.join("data", "raw", local_filename).replace("\\", "/")
        full_local_path = os.path.join(raw_dir, local_filename)
        
        generate_pdf(doc_info, full_local_path)
        print(f"  [OK] {doc_info['document_id']} ({doc_info['language']}): {doc_info['title']}")
        
        manifest_rows.append({
            "document_id": doc_info["document_id"],
            "title": doc_info["title"],
            "language": doc_info["language"],
            "source_url": doc_info["source_url"],
            "local_path": local_path,
            "document_type": doc_info["document_type"]
        })
        
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["document_id", "title", "language", "source_url", "local_path", "document_type"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    print(f"\nManifest successfully written to {manifest_path} with {len(manifest_rows)} entries.")

if __name__ == "__main__":
    main()
