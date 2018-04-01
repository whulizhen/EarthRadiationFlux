#ifndef GFC_GTIME_H
#define GFC_GTIME_H


//============================================================================
//
//  This file is part of GFC, the GNSS FOUNDATION CLASS.
//
//  The GFC is free software; you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published
//  by the Free Software Foundation; either version 3.0 of the License, or
//  any later version.
//
//  The GFC is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public
//  License along with GFC; if not, write to the Free Software Foundation,
//  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110, USA
//
//  Copyright 2015, lizhen
//
//============================================================================


#include "GException.h"
#include "GFCCONST.h"
#include <algorithm>
#include "GString.h"

namespace gfc
{
    NEW_GEXCEPTION_CLASS(InvalidTime,gfc::GException);
    
    NEW_GEXCEPTION_CLASS(TimeSystemUnexist, gfc::GException );
    
    /*ʱ��ϵͳ�Ķ���*/
    class TimeSystem  //ά��ʱ��ϵͳ�Ķ��弰ʱ��ϵͳ����
    {
        
    public:
        TimeSystem(void) { m_timeSystemName="tsUKN";}
        TimeSystem(GString timesysString)  { m_timeSystemName = timesysString;}
        virtual ~TimeSystem(void) {};
        
        /*
         ��̬��Ա���������ڶ�timesystemTab���г�ʼ��
         */
        
        static std::list<GString > Initializer();
        static void RegByName( GString variableName );
        
        static void UnregByName( GString variableName );
        static void dump( std::ostream& s ) ;
        
        static TimeSystem GetByName( GString variableName );
        
        TimeSystem& operator= (const TimeSystem& right)  //��ֵ����
        {
            this->m_timeSystemName = right.m_timeSystemName;
            return *this;
        }
        
        TimeSystem( const TimeSystem& right )   //�������캯��
        {
            this->m_timeSystemName = right.m_timeSystemName;
        }
        
        bool operator< (const TimeSystem& right) const //�Ⱥ�����
        {
            return this->m_timeSystemName < right.m_timeSystemName;
        }
        
        bool operator<= (const TimeSystem& right) const //�Ⱥ�����
        {
            return this->m_timeSystemName <= right.m_timeSystemName;
        }
        
        bool operator> (const TimeSystem& right) const //�Ⱥ�����
        {
            return this->m_timeSystemName > right.m_timeSystemName;
        }
        
        bool operator>= (const TimeSystem& right) const //�Ⱥ�����
        {
            return this->m_timeSystemName >= right.m_timeSystemName;
        }
        
        bool operator== (const TimeSystem& right) const //�Ⱥ�����
        {
            return this->m_timeSystemName == right.m_timeSystemName;
        }
        
        bool operator!= (const TimeSystem& right)  const //���Ⱥ�����
        {
            return !( operator==(right) );
        }
        
        // return the name(GString) of certain timesystem
        GString getTimeSystemName() { return m_timeSystemName;}
        
    private:
        
        static std::list<GString>  timesystemTab;
        
        GString m_timeSystemName;
    };
    
    //������ʱ��ϵͳ���ƻ�ȡʱ��ϵͳ�ĺ궨��
#define GTimeSystem(TimeSystemName) \
TimeSystem::GetByName(TimeSystemName)
    
    
    /*����ʱ����ʽ��������ʱ����*/
    class CivilTime
    {
        
    public:
        CivilTime() :m_year(0),m_month(0),m_day(0),m_hour(0),m_minute(0),m_second(0.0),m_ts(gfc::TimeSystem::GetByName("tsUKN")) {}
        ~CivilTime() {}
        
        GString TimeString()
        {
            GString timestr;
            char tmp[100]={0};
            sprintf(tmp, "%4d/%02d/%02d/%02d:%02d:%09.6f",m_year,m_month,m_day,m_hour,m_minute,m_second);
            timestr = tmp;
            return timestr;
        }
        
        CivilTime(GString sys, GString timestr)
        {
            std::vector<GString> split = timestr.split();
            m_ts = gfc::TimeSystem::GetByName(sys);
            m_year = split[0].asINT();
            m_month = split[1].asINT();
            m_day = split[2].asINT();
            m_hour = split[3].asINT();
            m_minute = split[4].asINT();
            m_second = split[5].asDOUBLE();
            
        }
        
        CivilTime( int year,int month,int day, int hour, int minute, double second,GString tStr)
        {
            m_year = year;
            m_month = month;
            m_day = day;
            m_hour = hour;
            m_minute = minute;
            m_second = second;
            m_ts = gfc::TimeSystem::GetByName(tStr);
        }
        
        int m_year;
        int m_month;
        int m_day;
        int m_hour;
        int m_minute;
        double m_second; //��λ����
        TimeSystem m_ts;
    };
    
    
    /*�����ʱ����ʽ*/
    class DOYTime
    {
        
    public:
        DOYTime() :m_year(0),m_doy(0),m_sod(0.0),m_ts(gfc::TimeSystem::GetByName("tsUKN")) {}
        ~DOYTime() {}
        DOYTime(int year, long day, double sod, GString tstr)
        {
            m_year = year;
            m_doy = day;
            m_sod = sod;
            m_ts = gfc::TimeSystem::GetByName(tstr);
        }
        int m_year;
        long m_doy;  //day of year
        double m_sod; //������,��λ����
        TimeSystem  m_ts;
    };
    
    
    /*����ʱ�䣬һ�����ܺ��������ʾ*/
    class NavTime
    {
        
    public:
        NavTime(): m_week(0),m_sow(0.0),m_ts(gfc::TimeSystem::GetByName("tsUKN")) {}
        ~NavTime() {}
        NavTime( int week, double sow, GString tstr)
        {
            m_week = week;
            m_sow = sow;
            m_ts = gfc::TimeSystem::GetByName(tstr);
        }
        
        //get day of week
        int getDOW()
        {
            int dow = -1;
            
            dow = int( m_sow/86400.0  ) ;
            
            return dow;
        }
        
        double getSOD()
        {
            double sod = -1;
            
            int dow =-1;
            
            dow = int( m_sow/86400.0  ) ;
            
            sod = m_sow - dow*86400.0;
            
            return sod;
        }
        
        int					 m_week;     //�ܼ���
        double				 m_sow;   //������
        TimeSystem  m_ts;   //ʱ��ϵͳ��Ŀǰֻ��GPST��BDT��GALT���ܺ��������ʾ
    };
    
    
    class JDTime
    {
        
    public:
        JDTime(): m_jd(0),m_sod(0.0),m_fsod(0.0),m_ts(gfc::TimeSystem::GetByName("tsUKN")) {}
        ~JDTime() {}
        JDTime(long jd, long sod, double fsod,GString tsStr)
        {
            m_jd = jd;
            m_sod = sod;
            m_fsod = fsod;
            m_ts = TimeSystem::GetByName(tsStr);
        }
        double jdt()
        {
            double secpday = 86400.0;
            secpday = secpday*1.0;
            long double jd = static_cast<long double>(m_sod + m_fsod)/secpday;
            jd = static_cast<long double>(m_jd) + jd;
            return  jd;
        }
        
        long m_jd;
        long m_sod;
        double m_fsod;
        TimeSystem m_ts;
    };
    
    
    
    
    //
    // Methods to handle time system conversion-------------------------------
    //
    //
    //          -14s
    //    -----------------> BDT(Compass Time)
    //    |
    //    |         +19s             +32.184s           +rel.effects
    //   GPST ------------> TAI ----------------> TT -----------------> TDB(TTB)
    //                      T |
    //           -(UT1-TAI) | |    -leap seconds
    //   UT1 ---------------| |--------------------> UTC
    //    |
    //    |   earth rotation
    //    ---------------------> GAST
    //========================================================================
    
				
    class GTime
    {
        
        
        
    public:
        
        /*�������ݽṹ�壬�����ڲ�ʹ��*/
        struct LeapType
        {
            
        public:
            int year;
            int month;
            int day;
            int nleap;
        };
        
        static std::vector<  LeapType >  InitializeLeapSecTable();
        //���������¼���������
        static	void  AddLeapSecond( int year, int month ,int day, int leaps );
        /*��ȡ����ĺ���Ҫ�Ǿ�̬��*/
        static double getLeapSecond(  int year, int month,int day);
        
        //��̬���ߺ���, time formats transformation
        static DOYTime CivilTime2DOYTime(CivilTime ct);
        static JDTime CivilTime2JDTime(CivilTime ct);
        static CivilTime JDTime2CivilTime(JDTime jdt);
        static GTime  JDTime2GTime(JDTime jdt);
        static JDTime GTime2JDTime(GTime gt);
        static CivilTime GTime2CivilTime(GTime gt);
        static GTime  CivilTime2GTime(CivilTime ct);
        static double JDCenturySince2000( GTime gt );
        static NavTime GTime2NavTime(GTime gt);
        
        
        //the definition of J2000 epoch
        //J2000 = 2451545.0 TT,
        //J2000 = January 1, 2000, 11:59:27.816 TAI
        //J2000 = January 1, 2000, 11:58:55.816 UTC
        static GTime J2000();
        
        // the definiton of Beidou System Time
        static GTime BDT0();
        
        // the definiton of GPS System Time
        static GTime GPST0();
        
         /* Function to convert from UTC to sidereal time
        * @param t         Epoch
        *
        * @return sidereal time in hours.
        */
       static double UTC2SID(GTime UTC);
       
       // return TAI-UTC
       // the timesystem of parament UTC should be in UTC time system
       //https://hpiers.obspm.fr/eoppc/bul/bulc/UTC-TAI.history
        static double TAImUTC( GTime UTC );
        
        // return TT-TAI in seconds.
        static double TTmTAI();
        
        // reutrn TAI-GPST in seconds
        static double TAImGPST() ;
        
        static GTime TAI2TT(GTime TAI);
        
        //a barycentric time
        // relativity correction between TT and TDB
        // ref Astronomical Almanac B7
        static double TDBmTT( GTime TT ,double ut1mutc = 0.0 , double* geocentricPOS = NULL );
        static GTime  TT2TDB(GTime TT,double tdbmtt);
        static GTime  TDB2TT(GTime TDB,double tdbmtt);
        
        static GTime TT2TAI(GTime TT);
        static GTime TAI2UTC( GTime TAI );
        static GTime UTC2TAI( GTime UTC );
        static GTime BDT2GPST(GTime BDT);
        static GTime GPST2BDT(GTime GPST);
        //transform from GPST to UTC
        static GTime GPST2UTC(GTime GPST);
        static GTime UTC2GPST(GTime UTC);
        // we must get ut1mutc from outside data source.
        // ut1mutc should be in seconds
        static GTime UT12UTC(GTime UT1, double ut1mutc);
        static GTime UTC2UT1(GTime UTC, double ut1mutc);
        void SetData(TimeSystem ts, long mjdp,long sodp, double fsodp);
        void GetData(TimeSystem& ts, long& mjdp,long& sodp, double& fsodp);
        void SetTimeSystem(TimeSystem ts) {m_ts = ts;}
        
        double toSeconds();
        double toDays();
        
        // time formats transformation
        void SetFromCivilTime(CivilTime ct);
        void SetFromDoyTime(DOYTime dt);
        void SetFromNavTime(NavTime nt);
        
        //        //���������
        GTime  operator+ (const double& second) const;
        GTime  operator- (const double& second) const;
        
        GTime  operator+ (const GTime& right) const;
        GTime  operator- (const GTime& right) const;
        
        
        bool operator==( const GTime& right ) const;
        bool operator!=( const GTime& right ) const;
        bool operator<( const  GTime& right ) const;
        bool operator>( const  GTime& right ) const;
        bool operator<=( const GTime& right ) const;
        bool operator>=( const GTime& right ) const;
        
        TimeSystem getTimeSystem() ;
        long       getMJD() const;

        long       getSOD() const;
        double     getFSOD() const ;
      
        virtual ~GTime() {};
        
        GTime();
        GTime(const GTime& time);   //�������캯��
        GTime& operator= (const GTime& right);  //��ֵ����
        
        GTime(long mjd, long sod, double fsod,GString tsStr);
        
        static  double eps;   //�����ж�ʱ����ȵ�tolerance
        
    private:	
        
        /// Default tolerance for time equality in days.
        
        //�������Ҫ�ܹ���̬����
        static std::vector< struct LeapType > LeapTable;
        
        TimeSystem    m_ts;  //ʱ��ϵͳ�Ķ���
        long 		m_mjd;   //Modified Julian Day
        long		    m_sod;  //�����������(��������) 0 -> 86399
        double		m_fsod;  //���С������(��λ����) 0->10^9;double�ľ���ֻ��С�����6λ
        
    };
    
    
//    //���������
//    bool   operator== ( const GTime& left, const GTime& right);
//    bool   operator!= ( const GTime& left, const GTime& right);
//    bool   operator<  ( const GTime& left, const GTime& right);
//    bool   operator>  ( const GTime& left, const GTime& right);
//    bool   operator<= ( const GTime& left, const GTime& right);
//    bool   operator>= ( const GTime& left, const GTime& right);
    
    
}  //end of namespace gfc


#endif
